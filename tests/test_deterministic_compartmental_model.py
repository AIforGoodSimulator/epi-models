import os
from collections import defaultdict
from math import floor
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from epi_models import CampParams, DeterministicCompartmentalModelRunner

# TODO: add more unit tests of different functions within the compartment model rather than just testing on these results

TEST_RESULT_KEYS = [
    "do_nothing_baseline",
    "camp_baseline",
    "better_hygiene_intervention_result",
    "increase_icu_intervention_result",
    "increase_remove_high_risk_result",
    "better_isolation_intervention_result",
    "shielding_intervention_result",
]


def instantiate_runner(num_iterations):
    base_dir = Path(os.path.dirname(__file__)).parents[0]
    camp_params = CampParams.load_from_json(
        base_dir / "epi_models" / "config" / "sample_input.json"
    )
    runner = DeterministicCompartmentalModelRunner(
        camp_params, num_iterations=num_iterations
    )
    return runner


def group_runner_results(runner):
    do_nothing_baseline, camp_baseline = runner.run_baselines()
    (
        better_hygiene_intervention_result,
        increase_icu_intervention_result,
        increase_remove_high_risk_result,
        better_isolation_intervention_result,
        shielding_intervention_result,
    ) = runner.run_different_scenarios()
    result_set = dict()
    result_set["population_size"] = runner.model.population_size
    result_set["isolation_capacity"] = runner.camp_params.isolation_capacity
    result_set["do_nothing_baseline"] = do_nothing_baseline
    result_set["camp_baseline"] = camp_baseline
    result_set["generated_params_df"] = runner.generated_params_df
    result_set[
        "better_hygiene_intervention_result"
    ] = better_hygiene_intervention_result
    result_set["increase_icu_intervention_result"] = increase_icu_intervention_result
    result_set["increase_remove_high_risk_result"] = increase_remove_high_risk_result
    result_set[
        "better_isolation_intervention_result"
    ] = better_isolation_intervention_result
    result_set["shielding_intervention_result"] = shielding_intervention_result
    return result_set


@pytest.fixture(scope="module")
def runner_once():
    return instantiate_runner(1)


@pytest.fixture(scope="module")
def runner_multiple_times():
    return instantiate_runner(10)


@pytest.fixture(scope="module")
def runner_results_once(runner_once):
    return group_runner_results(runner_once)


@pytest.fixture(scope="module")
def runner_results_multiple_times(runner_multiple_times):
    return group_runner_results(runner_multiple_times)


def test_individual_age_compartments(runner_results_once):
    result_set = runner_results_once
    for test_key in TEST_RESULT_KEYS:
        for compartment in [
            "Susceptible",
            "Exposed",
            "Infected_symptomatic",
            "Infected_asymptomatic",
            "Recovered",
            "Hospitalised",
            "Critical",
            "Deaths",
            "Offsite",
            "Quarantined",
            "No_ICU_Care",
        ]:
            combined_sub_age_compartments = (
                result_set[test_key][str(compartment) + "_0_9"]
                + result_set[test_key][str(compartment) + "_10_19"]
                + result_set[test_key][str(compartment) + "_20_29"]
                + result_set[test_key][str(compartment) + "_30_39"]
                + result_set[test_key][str(compartment) + "_40_49"]
                + result_set[test_key][str(compartment) + "_50_59"]
                + result_set[test_key][str(compartment) + "_60_69"]
                + result_set[test_key][str(compartment) + "_70_above"]
            )
            assert_allclose(
                result_set[test_key][compartment].values,
                combined_sub_age_compartments.values,
            )


def test_individual_compartment(runner_results_once):
    result_set = runner_results_once
    camp_population = result_set["population_size"]
    compartments = [
        "Susceptible",
        "Exposed",
        "Infected_symptomatic",
        "Infected_asymptomatic",
        "Recovered",
        "Hospitalised",
        "Critical",
        "Deaths",
        "Offsite",
        "Quarantined",
        "No_ICU_Care",
    ]
    for test_key in TEST_RESULT_KEYS:
        sum_population = result_set[test_key][compartments].sum(axis=1)
        camp_population_array = np.full(sum_population.shape, camp_population)
        assert_allclose(
            sum_population,
            camp_population_array,
            rtol=1e-06,
            err_msg=f"the {test_key} has unequal population number across the compartments",
        )


def test_intervention_better_hygiene(runner_results_multiple_times):
    result_set = runner_results_multiple_times
    time_range_cutoff = {
        "one_month": 31,
        "three_month": 61,
        "six_month": 71,
    }  # surprised by how quickly common sense decayed here...
    effectiveness_range = ["5%", "10%", "15%"]
    do_nothing_baseline = result_set["do_nothing_baseline"]
    do_nothing_mean = do_nothing_baseline.groupby("Time").mean()
    better_hygiene_intervention_result = result_set[
        "better_hygiene_intervention_result"
    ]
    better_hygiene_intervention_result_mean = defaultdict(dict)
    for time_range in time_range_cutoff:
        for effectiveness in effectiveness_range:
            better_hygiene_intervention_result_mean[time_range][effectiveness] = (
                better_hygiene_intervention_result[
                    better_hygiene_intervention_result["Scenario_suffix"]
                    == "|".join([effectiveness, time_range])
                ]
                .groupby("Time")
                .mean()
            )
    for time_range in time_range_cutoff:
        for compartment in [
            "Exposed",
            "Infected_symptomatic",
            "Infected_asymptomatic",
            "Hospitalised",
        ]:
            zero_percent = do_nothing_mean.loc[
                1 : time_range_cutoff[time_range], compartment
            ].values
            five_percent = (
                better_hygiene_intervention_result_mean[time_range]["5%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            ten_percent = (
                better_hygiene_intervention_result_mean[time_range]["10%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            fifteen_percent = (
                better_hygiene_intervention_result_mean[time_range]["15%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            assert_array_less(
                five_percent,
                zero_percent,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between 5% and baseline",
            )
            assert_array_less(
                ten_percent,
                five_percent,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between 10% and 5%",
            )
            assert_array_less(
                fifteen_percent,
                ten_percent,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between 15% and 10%",
            )


def test_intervention_isolation(runner_results_multiple_times):
    result_set = runner_results_multiple_times
    do_nothing_baseline = result_set["do_nothing_baseline"]
    do_nothing_mean = do_nothing_baseline.groupby("Time").mean()
    isolation_capacity = result_set["isolation_capacity"]
    lower_bound_capacity, upper_bound_capacity = [
        isolation_capacity,
        floor(isolation_capacity * 1.5),
    ]
    capacity_range = ["low_bound", "upper_bound"]
    rate_range = ["0.05%", "0.1%", "0.25%"]
    time_range_cutoff = {"fifty_day": 51, "one_hundred_day": 90, "two_hundred_day": 90}
    better_iso_intervention_result = result_set["better_isolation_intervention_result"]
    better_iso_intervention_result_mean = defaultdict(dict)
    for time_range in time_range_cutoff:
        for capacity_name in capacity_range:
            better_iso_intervention_result_mean[time_range][capacity_name] = dict()
            for rate_name in rate_range:
                scenario_id = "|".join([capacity_name, rate_name, time_range])
                better_iso_intervention_result_mean[time_range][capacity_name][
                    rate_name
                ] = (
                    better_iso_intervention_result[
                        better_iso_intervention_result["Scenario_suffix"] == scenario_id
                    ]
                    .groupby("Time")
                    .mean()
                )
    for time_range in time_range_cutoff:
        for compartment in [
            "Infected_symptomatic",
            "Infected_asymptomatic",
            "Hospitalised",
            "Deaths",
        ]:
            baseline = do_nothing_mean.loc[
                1 : time_range_cutoff[time_range], compartment
            ].values
            lower_bound_lower_rate = (
                better_iso_intervention_result_mean[time_range]["low_bound"]["0.05%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            upper_bound_lower_rate = (
                better_iso_intervention_result_mean[time_range]["upper_bound"]["0.05%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            lower_bound_medium_rate = (
                better_iso_intervention_result_mean[time_range]["low_bound"]["0.1%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            upper_bound_medium_rate = (
                better_iso_intervention_result_mean[time_range]["upper_bound"]["0.1%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            lower_bound_higher_rate = (
                better_iso_intervention_result_mean[time_range]["low_bound"]["0.25%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            upper_bound_higher_rate = (
                better_iso_intervention_result_mean[time_range]["upper_bound"]["0.25%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )

            assert_array_less(
                lower_bound_lower_rate,
                baseline,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between lower bound lower rate and baseline",
            )
            assert_array_less(
                lower_bound_medium_rate,
                baseline,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between lower bound medium rate and lower bound lower rate",
            )
            assert_array_less(
                lower_bound_higher_rate,
                baseline,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between lower bound higher rate and lower bound medium rate",
            )
            assert_array_less(
                upper_bound_lower_rate,
                baseline,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between upper bound lower rate and baseline",
            )
            assert_array_less(
                upper_bound_medium_rate,
                baseline,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between upper bound medium rate and upper bound lower rate",
            )
            assert_array_less(
                upper_bound_higher_rate,
                baseline,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values between upper bound higher rate and upper bound medium rate",
            )
        for compartment in ["Quarantined"]:
            lower_bound_lower_rate = (
                better_iso_intervention_result_mean[time_range]["low_bound"]["0.05%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            upper_bound_lower_rate = (
                better_iso_intervention_result_mean[time_range]["upper_bound"]["0.05%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            lower_bound_medium_rate = (
                better_iso_intervention_result_mean[time_range]["low_bound"]["0.1%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            upper_bound_medium_rate = (
                better_iso_intervention_result_mean[time_range]["upper_bound"]["0.1%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            lower_bound_higher_rate = (
                better_iso_intervention_result_mean[time_range]["low_bound"]["0.25%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            upper_bound_higher_rate = (
                better_iso_intervention_result_mean[time_range]["upper_bound"]["0.25%"]
                .loc[1 : time_range_cutoff[time_range], compartment]
                .values
            )
            lower_bound_capacity_array = np.array(
                [lower_bound_capacity] * len(lower_bound_lower_rate)
            )
            upper_bound_capacity_array = np.array(
                [upper_bound_capacity] * len(upper_bound_lower_rate)
            )
            assert_array_less(
                lower_bound_lower_rate,
                lower_bound_capacity_array,
                err_msg=f"{compartment} with time cutoff {time_range} has exceeded the capacity limit",
            )
            assert_array_less(
                lower_bound_medium_rate,
                lower_bound_capacity_array,
                err_msg=f"{compartment} with time cutoff {time_range} has exceeded the capacity limit",
            )
            assert_array_less(
                lower_bound_higher_rate,
                lower_bound_capacity_array,
                err_msg=f"{compartment} with time cutoff {time_range} has exceeded the capacity limit",
            )
            assert_array_less(
                upper_bound_lower_rate,
                upper_bound_capacity_array,
                err_msg=f"{compartment} with time cutoff {time_range} has exceeded the capacity limit",
            )
            assert_array_less(
                upper_bound_medium_rate,
                upper_bound_capacity_array,
                err_msg=f"{compartment} with time cutoff {time_range} has exceeded the capacity limit",
            )
            assert_array_less(
                upper_bound_higher_rate,
                upper_bound_capacity_array,
                err_msg=f"{compartment} with time cutoff {time_range} has exceeded the capacity limit",
            )


def test_intervention_remove_high_risk(runner_results_multiple_times):
    result_set = runner_results_multiple_times
    do_nothing_baseline = result_set["do_nothing_baseline"]
    do_nothing_mean = do_nothing_baseline.groupby("Time").mean()
    time_range_cutoff = {
        "removal_one_week": 8,
        "removal_three_week": 22,
        "removal_six_week": 43,
    }
    increase_remove_high_risk_result = result_set["increase_remove_high_risk_result"]
    increase_remove_high_risk_result_mean = defaultdict(dict)
    for time_range in time_range_cutoff:
        increase_remove_high_risk_result_mean[time_range] = (
            increase_remove_high_risk_result[
                increase_remove_high_risk_result["Scenario_suffix"] == time_range
            ]
            .groupby("Time")
            .mean()
        )
    # comparing offsite/hospitalisation/death
    for time_range in time_range_cutoff:
        for compartment in ["Hospitalised", "No_ICU_Care", "Deaths"]:
            baseline = do_nothing_mean.loc[
                time_range_cutoff[time_range] : 170, compartment
            ].values
            remove_high_risk = (
                increase_remove_high_risk_result_mean[time_range]
                .loc[time_range_cutoff[time_range] : 170, compartment]
                .values
            )
            assert_array_less(
                remove_high_risk,
                baseline,
                err_msg=f"{compartment} with time cutoff {time_range} has unequal values remove high risk residents and baseline",
            )
    remove_high_risk_one_week = (
        increase_remove_high_risk_result_mean["removal_one_week"]
        .loc[time_range_cutoff[time_range] :, "Offsite"]
        .values
    )
    remove_high_risk_three_week = (
        increase_remove_high_risk_result_mean["removal_one_week"]
        .loc[time_range_cutoff[time_range] :, "Offsite"]
        .values
    )
    remove_high_risk_six_week = (
        increase_remove_high_risk_result_mean["removal_one_week"]
        .loc[time_range_cutoff[time_range] :, "Offsite"]
        .values
    )
    assert_allclose(remove_high_risk_one_week, remove_high_risk_three_week)
    assert_allclose(remove_high_risk_one_week, remove_high_risk_six_week)


def test_intervention_increase_icu(runner_results_multiple_times):
    result_set = runner_results_multiple_times
    camp_baseline = result_set["camp_baseline"]
    camp_baseline_mean = camp_baseline.groupby("Time").mean()
    ideal_icu_capacity = result_set["population_size"] * 0.001
    increase_icu_result = result_set["increase_icu_intervention_result"]
    increase_icu_result_mean = (
        increase_icu_result[
            increase_icu_result["Scenario_suffix"] == "increase_to_ideal_icu_capacity"
        ]
        .groupby("Time")
        .mean()
    )
    # test death and critical
    baseline_critical_number = camp_baseline_mean.loc[41:, "Critical"].values
    critical_capacity_array = np.array(
        [ideal_icu_capacity] * len(baseline_critical_number)
    )
    icu_critical_number = increase_icu_result_mean.loc[41:, "Critical"].values
    assert_array_less(icu_critical_number, critical_capacity_array)
    assert_array_less(baseline_critical_number, icu_critical_number)
    baseline_deaths_number = camp_baseline_mean.loc[41:, "Deaths"].values
    icu_deaths_number = increase_icu_result_mean.loc[41:, "Deaths"].values
    assert_array_less(icu_deaths_number, baseline_deaths_number)


def test_intervention_shielding(runner_results_multiple_times):
    pass


# def test_intervention_isolation(runner_results_multiple_times):
#     result_set = runner_results_multiple_times
#     do_nothing_baseline = result_set["do_nothing_baseline"]
#     capacity_range = ["low_bound", "upper_bound"]
#     rate_range = ["0.05%", "0.1%", "0.25%"]
#     time_range_cutoff = {"fifty_day": 51, "one_hundred_day": 101, "two_hundred_day": 201}
#     scenario_id = "|".join([capacity_name, rate_name, end_times_name])
#     iso_6_month_results_groups = iso_6_month_results.groupby("R0")
#     do_nothing_infected = 0
#     for index, group in sim_groups:
#         do_nothing_infected = group["Infected_symptomatic"][1:31]
#     do_nothing_hospitalised = 0
#     for index, group in sim_groups:
#         do_nothing_hospitalised = group["Hospitalised"][1:31]
#     do_nothing_critical = 0
#     for index, group in sim_groups:
#         do_nothing_critical = group["Critical"][1:31]
#     do_nothing_deaths = 0
#     for index, group in sim_groups:
#         do_nothing_deaths = group["Deaths"][1:31]
#     isolation_infected = 0
#     for index, group in iso_6_month_results_groups:
#         isolation_infected = group["Infected_symptomatic"][1:31]
#
#     isolation_hospitalised = 0
#     for index, group in iso_6_month_results_groups:
#         isolation_hospitalised = group["Hospitalised"][1:31]
#
#     isolation_critical = 0
#     for index, group in iso_6_month_results_groups:
#         isolation_critical = group["Critical"][1:31]
#
#     isolation_deaths = 0
#     for index, group in iso_6_month_results_groups:
#         isolation_deaths = group["Deaths"][1:31]
#     # testing infected is lesser in isolation scenarios than do nothing
#     assert_array_less(isolation_infected, do_nothing_infected)
#
#     # testing hospitalised is lesser in isolation scenarios than do nothing
#     assert_array_less(isolation_hospitalised, do_nothing_hospitalised)
#
#     # testing critical is lesser in isolation scenarios than do nothing
#     assert_array_less(isolation_critical, do_nothing_critical)
#
#     # testing deaths is lesser in isolation scenarios than do nothing
#     assert_array_less(isolation_deaths, do_nothing_deaths)
#
#
# def test_camp_baselines_with_do_nothing(instantiate_runner):
#     result_set = instantiate_runner
#     do_nothing_baseline = result_set["do_nothing_baseline"]
#     camp_baseline = result_set["camp_baseline"]
#     sim_groups = do_nothing_baseline.groupby("R0")
#     sim_groups_camp = camp_baseline.groupby("R0")
#     do_nothing_infected = 0
#     for index, group in sim_groups:
#         do_nothing_infected = group["Infected_symptomatic"][1:31]
#     do_nothing_hospitalised = 0
#     for index, group in sim_groups:
#         do_nothing_hospitalised = group["Hospitalised"][1:31]
#     do_nothing_critical = 0
#     for index, group in sim_groups:
#         do_nothing_critical = group["Critical"][1:31]
#     do_nothing_deaths = 0
#     for index, group in sim_groups:
#         do_nothing_deaths = group["Deaths"][1:31]
#     camp_baseline_infected = 0
#     for index, group in sim_groups_camp:
#         camp_baseline_infected = group["Infected_symptomatic"][1:31]
#     camp_baseline_hospitalised = 0
#     for index, group in sim_groups_camp:
#         camp_baseline_hospitalised = group["Hospitalised"][1:31]
#     camp_baseline_critical = 0
#     for index, group in sim_groups_camp:
#         camp_baseline_critical = group["Critical"][1:31]
#     camp_baseline_deaths = 0
#     for index, group in sim_groups_camp:
#         camp_baseline_deaths = group["Deaths"][1:31]
#     # testing infected is lesser in isolation scenarios than do nothing
#     assert_array_less(camp_baseline_infected, do_nothing_infected)
#
#     # testing hospitalised is lesser in isolation scenarios than do nothing
#     assert_array_less(camp_baseline_hospitalised, do_nothing_hospitalised)
#
#     # testing critical is lesser in isolation scenarios than do nothing
#     assert_array_less(camp_baseline_critical, do_nothing_critical)
#
#     # testing deaths is lesser in isolation scenarios than do nothing
#     assert_array_less(camp_baseline_deaths, do_nothing_deaths)
#
#
# def test_run_different_scenarios(instantiate_runner):
#     # TODO: groupby take the mean
#     result_set = instantiate_runner
#     do_nothing_baseline = result_set["do_nothing_baseline"]
#     better_hygiene_intervention_result = result_set[
#         "better_hygiene_intervention_result"
#     ]
#     increase_icu_intervention_result = result_set["increase_icu_intervention_result"]
#     increase_remove_high_risk_result = result_set["increase_remove_high_risk_result"]
#     better_isolation_intervention_result = result_set[
#         "better_isolation_intervention_result"
#     ]
#     shielding_intervention_result = result_set["shielding_intervention_result"]
#     sim_groups = do_nothing_baseline.groupby("R0")
#     sim_group_better_hygiene = better_hygiene_intervention_result.groupby("R0")
#     sim_group_increase_icu = increase_icu_intervention_result.groupby("R0")
#     sim_group_increase_remove_high_risk = increase_remove_high_risk_result.groupby("R0")
#     sim_group_better_isolation = better_isolation_intervention_result.groupby("R0")
#     sim_group_shielding_intervention = shielding_intervention_result.groupby("R0")
#     do_nothing_infected = 0
#     for index, group in sim_groups:
#         do_nothing_infected = group["Infected_symptomatic"][1:31]
#     do_nothing_hospitalised = 0
#     for index, group in sim_groups:
#         do_nothing_hospitalised = group["Hospitalised"][1:31]
#     do_nothing_critical = 0
#     for index, group in sim_groups:
#         do_nothing_critical = group["Critical"][1:31]
#     do_nothing_uncared = 0
#     for index, group in sim_groups:
#         do_nothing_uncared = group["No_ICU_Care"][1:31]
#     do_nothing_deaths = 0
#     for index, group in sim_groups:
#         do_nothing_deaths = group["Deaths"][1:31]
#     better_hygiene_infected = 0
#     for index, group in sim_group_better_hygiene:
#         better_hygiene_infected = group["Infected_symptomatic"][1:31]
#     better_hygiene_hospitalised = 0
#     for index, group in sim_group_better_hygiene:
#         better_hygiene_hospitalised = group["Hospitalised"][1:31]
#     better_hygiene_critical = 0
#     for index, group in sim_group_better_hygiene:
#         better_hygiene_critical = group["Critical"][1:31]
#     better_hygiene_deaths = 0
#     for index, group in sim_group_better_hygiene:
#         better_hygiene_deaths = group["Deaths"][1:31]
#     increase_icu_infected = 0
#     for index, group in sim_group_increase_icu:
#         increase_icu_infected = group["Infected_symptomatic"][1:31]
#     increase_icu_hospitalised = 0
#     for index, group in sim_group_increase_icu:
#         increase_icu_hospitalised = group["Hospitalised"][1:31]
#     increase_icu_critical = 0
#     for index, group in sim_group_increase_icu:
#         increase_icu_critical = group["Critical"][1:31]
#     increase_icu_uncared = 0
#     for index, group in sim_group_increase_icu:
#         increase_icu_uncared = group["No_ICU_Care"][1:31]
#     increase_icu_deaths = 0
#     for index, group in sim_group_increase_icu:
#         increase_icu_deaths = group["Deaths"][1:31]
#     increase_remove_high_risk_infected = 0
#     for index, group in sim_group_increase_remove_high_risk:
#         increase_remove_high_risk_infected = group["Infected_symptomatic"][1:31]
#     increase_remove_high_risk_hospitalised = 0
#     for index, group in sim_group_increase_remove_high_risk:
#         increase_remove_high_risk_hospitalised = group["Hospitalised"][1:31]
#     increase_remove_high_risk_critical = 0
#     for index, group in sim_group_increase_remove_high_risk:
#         increase_remove_high_risk_critical = group["Critical"][1:31]
#     increase_remove_high_risk_deaths = 0
#     for index, group in sim_group_increase_remove_high_risk:
#         increase_remove_high_risk_deaths = group["Deaths"][1:31]
#     better_isolation_infected = 0
#     for index, group in sim_group_better_isolation:
#         better_isolation_infected = group["Infected_symptomatic"][1:31]
#     better_isolation_hospitalised = 0
#     for index, group in sim_group_better_isolation:
#         better_isolation_hospitalised = group["Hospitalised"][1:31]
#     better_isolation_critical = 0
#     for index, group in sim_group_better_isolation:
#         better_isolation_critical = group["Critical"][1:31]
#     better_isolation_deaths = 0
#     for index, group in sim_group_better_isolation:
#         better_isolation_deaths = group["Deaths"][1:31]
#     # shielding_intervention_infected = 0
#     # for index, group in sim_group_shielding_intervention:
#     #     shielding_intervention_infected = group["Infected_symptomatic"][1:31]
#     # shielding_intervention_hospitalised = 0
#     # for index, group in sim_group_shielding_intervention:
#     #     shielding_intervention_hospitalised = group["Hospitalised"][1:31]
#     # shielding_intervention_critical = 0
#     # for index, group in sim_group_shielding_intervention:
#     #     shielding_intervention_critical = group["Critical"][1:31]
#     # shielding_intervention_deaths = 0
#     # for index, group in sim_group_shielding_intervention:
#     #     shielding_intervention_deaths = group["Deaths"][1:31]
#
#     # testing infected is lesser in isolation scenarios than do nothing
#     assert_array_less(better_hygiene_infected, do_nothing_infected)
#
#     # testing hospitalised is lesser in isolation scenarios than do nothing
#     assert_array_less(better_hygiene_hospitalised, do_nothing_hospitalised)
#
#     # testing critical is lesser in isolation scenarios than do nothing
#     assert_array_less(better_hygiene_critical, do_nothing_critical)
#
#     # testing deaths is lesser in isolation scenarios than do nothing
#     assert_array_less(better_hygiene_deaths, do_nothing_deaths)
#
#     # # testing infected is lesser in increasing ICU scenarios than do nothing
#     # assert_array_less(increase_icu_infected, do_nothing_infected)
#     #
#     # # testing hospitalised is lesser in increasing ICU scenarios than do nothing
#     # assert_array_less(increase_icu_hospitalised, do_nothing_hospitalised)
#
#     # testing critical is more in increasing ICU scenarios than do nothing
#     assert_array_less(do_nothing_critical, increase_icu_critical)
#
#     # testing uncared is less in increasing ICU scenarios than do nothing
#     assert_array_less(increase_icu_uncared, do_nothing_uncared)
#
#     # testing deaths is lesser in increasing ICU scenarios than do nothing
#     assert_array_less(increase_icu_deaths, do_nothing_deaths)
#
#     # testing infected is lesser in isolation scenarios than do nothing
#     assert_array_less(increase_remove_high_risk_infected, do_nothing_infected)
#
#     # testing hospitalised is lesser in isolation scenarios than do nothing
#     assert_array_less(increase_remove_high_risk_hospitalised, do_nothing_hospitalised)
#
#     # testing critical is lesser in isolation scenarios than do nothing
#     assert_array_less(increase_remove_high_risk_critical, do_nothing_critical)
#
#     # testing deaths is lesser in isolation scenarios than do nothing
#     assert_array_less(increase_remove_high_risk_deaths, do_nothing_deaths)
#
#     # testing infected is lesser in isolation scenarios than do nothing
#     assert_array_less(better_isolation_infected, do_nothing_infected)
#
#     # testing hospitalised is lesser in isolation scenarios than do nothing
#     assert_array_less(better_isolation_hospitalised, do_nothing_hospitalised)
#
#     # testing critical is lesser in isolation scenarios than do nothing
#     assert_array_less(better_isolation_critical, do_nothing_critical)
#
#     # testing deaths is lesser in isolation scenarios than do nothing
#     assert_array_less(better_isolation_deaths, do_nothing_deaths)
#
#     # # testing infected is lesser in isolation scenarios than do nothing
#     # assert_array_less(shielding_intervention_infected, do_nothing_infected)
#
#     # # testing hospitalised is lesser in isolation scenarios than do nothing
#     # assert_array_less(shielding_intervention_hospitalised, do_nothing_hospitalised)
#
#     # # testing critical is lesser in isolation scenarios than do nothing
#     # assert_array_less(shielding_intervention_critical, do_nothing_critical)
#
#     # # testing deaths is lesser in isolation scenarios than do nothing
#     # assert_array_less(shielding_intervention_deaths, do_nothing_deaths)
