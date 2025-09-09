import numpy as np
from prettytable import PrettyTable
import plotly.graph_objects as go
from utils import get_data


def generate_chart_distance(errors_list, name, labels, filter=False):
    fig = go.Figure()
    colors = ['green', 'purple', 'red', 'orange']

    for errors, label, color in zip(errors_list, labels, colors):
        if filter:
            errors = errors[errors < 10000]

        sorted_errors = np.sort(errors)
        quotients = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        fig.add_trace(go.Scatter(
            x=sorted_errors, y=quotients,
            mode='lines+markers',
            name=label,
            line=dict(color=color)
        ))

    fig.update_layout(
        title=name,
        xaxis_title='Vzdialenosť od správneho bodu v milimetroch',
        yaxis_title='Podiel prvkoch s chybou menšou ako x',
        xaxis=dict(type='log', tickformat='.0f'),
        template='plotly_white'
    )
    return fig


def generate_chart_angle(errors_list, name, labels, filter=False):
    fig = go.Figure()
    colors = ['green', 'purple', 'red', 'orange']

    for errors, label, color in zip(errors_list, labels, colors):
        if filter:
            errors = errors[errors < 60]

        sorted_errors = np.sort(errors)
        quotients = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        fig.add_trace(go.Scatter(
            x=sorted_errors, y=quotients,
            mode='lines+markers',
            name=label,
            line=dict(color=color)
        ))

    fig.update_layout(
        title=name,
        xaxis_title='Veľkosť uhlovej chyby',
        yaxis_title='Podiel prvkoch s chybou menšou ako x',
        xaxis=dict(tickformat='.0f'),
        template='plotly_white'
    )
    return fig


def mean_error_distance(errs, field_names):
    last = 10000
    for array in errs:
        array[np.isnan(array)] = 100000

    tab = PrettyTable()
    tab.field_names = field_names

    labels = ['L2CS', 'GazeTR', 'L2CS_cropped', 'GazeTR_cropped']
    precision = 1
    for label, error in zip(labels, errs):
        error = error[error < last]
        res = np.array([np.sum(error < t) / len(error) for t in range(1, last + 1)])

        median_err = np.round(np.median(error), 2)
        mean_err = np.round(np.mean(error), 2)
        mean_res_100 = np.round(np.mean(res[:100]) * 100, precision)
        mean_res_1000 = np.round(np.mean(res[:1000]) * 100, precision)
        mean_res_10000 = np.round(np.mean(res) * 100, precision)

        tab.add_row([label, median_err, mean_err, mean_res_100, mean_res_1000, mean_res_10000])
    tab.sortby = "Sieť"
    return tab


def mean_error_angle(errs, field_names):
    for array in errs:
        array[np.isnan(array)] = 180

    tab = PrettyTable()
    tab.field_names = field_names

    last = 40
    labels = ['L2CS', 'GazeTR', 'L2CS_cropped', 'GazeTR_cropped']
    for label, error in zip(labels, errs):
        res = np.array([np.sum(error < t) / len(error) for t in range(1, last)])
        precision = 1
        median_err = np.round(np.median(error), 2)
        mean_err = np.round(np.mean(error), 2)
        mean_res_1_5 = np.round(np.mean(res[:5]) * 100, precision)
        mean_res_1_10 = np.round(np.mean(res[:10]) * 100, precision)
        mean_res_1_20 = np.round(np.mean(res[:20]) * 100, precision)
        mean_res_1_30 = np.round(np.mean(res[:30]) * 100, precision)
        mean_res_1_40 = np.round(np.mean(res) * 100, precision)

        tab.add_row(
            [label, median_err, mean_err, mean_res_1_5, mean_res_1_10, mean_res_1_20, mean_res_1_30, mean_res_1_40])
    tab.sortby = "Sieť"
    return tab


def add_line_to_chart(fig, errors, label, color, marker='circle'):
    if filter:
        errors = errors[errors < 60]

    sorted_errors = np.sort(errors)
    quotients = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    fig.add_trace(go.Scatter(
        x=sorted_errors, y=quotients,
        mode='lines+markers',
        name=label,
        line=dict(color=color),
        marker=dict(symbol=marker)
    ))
    return fig


def add_column_to_table(tab, error, last, label, isAngle=False, mark=True):
    res = np.array([np.sum(error < t) / len(error) for t in range(1, last)])
    precision = 1

    median_err = np.round(np.median(error), 2)
    mean_err = np.round(np.mean(error), 2)
    if isAngle:
        mean_res_1_5 = np.round(np.mean(res[:5]) * 100, precision)
        mean_res_1_10 = np.round(np.mean(res[:10]) * 100, precision)
        mean_res_1_20 = np.round(np.mean(res[:20]) * 100, precision)
        mean_res_1_30 = np.round(np.mean(res[:30]) * 100, precision)
        mean_res_1_40 = np.round(np.mean(res) * 100, precision)

        tab.add_row(
            [label, mark, median_err, mean_err, mean_res_1_5, mean_res_1_10, mean_res_1_20, mean_res_1_30, mean_res_1_40])


    else:
        mean_res_100 = np.round(np.mean(res[:100]) * 100, precision)
        mean_res_1000 = np.round(np.mean(res[:1000]) * 100, precision)
        mean_res_10000 = np.round(np.mean(res) * 100, precision)

        tab.add_row([label, mark, median_err, mean_err, mean_res_100, mean_res_1000, mean_res_10000])

    return tab


def mean_error_glasses(errors_with_glasses, errors_no_glasses, field_names, labels, isAngle=False):
    last = None
    if isAngle:
        for array, array2 in zip(errors_with_glasses, errors_no_glasses):
            array[np.isnan(array)] = 180
            array2[np.isnan(array2)] = 180
            last = 40
    else:
        for array, array2 in zip(errors_with_glasses, errors_no_glasses):
            array[np.isnan(array)] = 100000
            array2[np.isnan(array2)] = 100000
            last = 10000
    tab = PrettyTable()
    tab.field_names = field_names

    for label, error in zip(labels, errors_with_glasses):
        # label += ' w_g'
        tab = add_column_to_table(tab, error, last, label, isAngle=isAngle, mark=True)

    for label, error in zip(labels, errors_no_glasses):
        # label += ' w/o_g'
        tab = add_column_to_table(tab, error, last, label, isAngle=isAngle, mark=False)

    tab.sortby = "Sieť"
    return tab


def generate_chart_compare_angle(errors_with_glasses, errors_no_glasses, name, labels, filter=False):
    fig = go.Figure()
    colors = ['green', 'purple', 'red', 'orange']

    for errors, label, color in zip(errors_with_glasses, labels, colors):
        label += ' w_g'
        fig = add_line_to_chart(fig, errors, label, color, marker='circle')

    for errors, label, color in zip(errors_no_glasses, labels, colors):
        label += ' w/o_g'
        fig = add_line_to_chart(fig, errors, label, color, marker='diamond')

    # Adding labels and title
    fig.update_layout(
        title=name,
        xaxis_title='Veľkosť uhlovej chyby',
        yaxis_title='Podiel prvkoch s chybou menšou ako x',
        xaxis=dict(tickformat='.0f'),
        template='plotly_white'
    )
    return fig


def show_charts(save_dir):
    dirc_full = get_data(save_dir)
    dirc_cropped = get_data(save_dir, prefix="cropped")
    labels = ['L2CS', 'GazeTR', 'L2CS_cr', 'GazeTR_cr']

    errors_list = [dirc_full["combined"]["L2CS"]["distance"],
                   dirc_full["combined"]["GazeTR"]["distance"],
                   dirc_cropped["combined"]["L2CS"]["distance"],
                   dirc_cropped["combined"]["GazeTR"]["distance"]]
    field_name_distance = ["Sieť", "Medián", "Priemer", "AUC 10^2", "AUC 10^3", "AUC 10^4"]
    field_name_angle = ["Sieť", "Medián", "Priemer", "AUC 5", "AUC 10", "AUC 20,",
                        "AUC 30", "AUC 40"]
    fig_all_distance = generate_chart_distance(errors_list, "Vzdialenostná chyba", labels, filter=False)
    table_distance_error = mean_error_distance(errors_list, field_name_distance)

    errors_list = [dirc_full["combined"]["L2CS"]["angle"],
                   dirc_full["combined"]["GazeTR"]["angle"],
                   dirc_cropped["combined"]["L2CS"]["angle"],
                   dirc_cropped["combined"]["GazeTR"]["angle"]]
    fig_all_angle = generate_chart_angle(errors_list, "Uhlová chyba", labels, filter=True)
    table_angle_error = mean_error_angle(errors_list, field_name_angle)

    errors_list_glasses = [dirc_full["with_glasses"]["L2CS"]["angle"],
                           dirc_full["with_glasses"]["GazeTR"]["angle"],
                           dirc_cropped["with_glasses"]["L2CS"]["angle"],
                           dirc_cropped["with_glasses"]["GazeTR"]["angle"]]
    errors_list_no_glasses = [dirc_full["no_glasses"]["L2CS"]["angle"],
                              dirc_full["no_glasses"]["GazeTR"]["angle"],
                              dirc_cropped["no_glasses"]["L2CS"]["angle"],
                              dirc_cropped["no_glasses"]["GazeTR"]["angle"]]
    fig_glasses_angle = generate_chart_compare_angle(errors_list_glasses, errors_list_no_glasses,
                                                            "Uhlová chyba - dataset s okuliarmi", labels, filter=True)

    field_name_angle.insert(1, "Okuliare")
    table_glasses_angle = mean_error_glasses(errors_list_glasses, errors_list_no_glasses, field_name_angle, labels, isAngle=True)

    errors_list_glasses = [dirc_full["with_glasses"]["L2CS"]["distance"],
                           dirc_full["with_glasses"]["GazeTR"]["distance"],
                           dirc_cropped["with_glasses"]["L2CS"]["distance"],
                           dirc_cropped["with_glasses"]["GazeTR"]["distance"]]
    errors_list_no_glasses = [dirc_full["no_glasses"]["L2CS"]["distance"],
                              dirc_full["no_glasses"]["GazeTR"]["distance"],
                              dirc_cropped["no_glasses"]["L2CS"]["distance"],
                              dirc_cropped["no_glasses"]["GazeTR"]["distance"]]

    field_name_distance.insert(1, "Okuliare")
    table_glasses_distance = mean_error_glasses(errors_list_glasses, errors_list_no_glasses, field_name_distance, labels,
                                              isAngle=False)

    fig_all_distance.show()
    fig_all_angle.show()

    print(table_angle_error)
    print(table_distance_error)
    print(table_glasses_angle)
    print(table_glasses_distance)