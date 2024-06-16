:- dynamic feature/2.
:- dynamic class/2.
:- dynamic ground_truth/2.

:- use_module(library(http/json)).

% Convertir JSON a reglas Prolog
json_to_prolog_rules(JSON) :-
    retractall(feature(_, _)),
    retractall(class(_, _)),
    retractall(ground_truth(_, _)),
    % Extraer el árbol del JSON
    Tree = JSON.get(tree),
    Y_feature = JSON.get(y_feature),
    Y_feature_name = Y_feature.get(name),
    Y_feature_values = Y_feature.get(values),
    assertz(ground_truth(Y_feature_name, Y_feature_values)),
    parse_tree(Tree, root).

% Parsear el árbol y almacenar características y clases
parse_tree(Node, Path) :-
    atom_concat(Path, '_left', LeftPath),
    atom_concat(Path, '_right', RightPath),
    Feature = Node.get(feature_index),
    Threshold = Node.get(split_treshold),
    Impurity = Node.get(impurity),
    ImpurityType = Node.get(impurity_type),
    Samples = Node.get(samples),
    Values = Node.get(value),
    YImpurity = Node.get(y_impurity),

    assertz(feature(Path, feature(Feature, Threshold, Impurity, ImpurityType, Samples, Values, YImpurity))),
    handle_left(Node, LeftPath),
    handle_right(Node, RightPath).

% Manejar el hijo izquierdo del nodo
handle_left(Node, LeftPath) :-
    handle_child(Node, LeftPath, left).

% Manejar el hijo derecho del nodo
handle_right(Node, RightPath) :-
    handle_child(Node, RightPath, right).

% Manejar el nodo hijo (izquierdo o derecho)
handle_child(Node, ChildPath, ChildDirection) :-
    ChildNode = Node.get(ChildDirection),
    !,
    parse_tree(ChildNode, ChildPath).

% Almacenar la clase final de una hoja
handle_child(Node, Path, _) :-
    Values = Node.get(value),
    assertz(class(Path, Values)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Predicción basada en el árbol construido
predict(X_value, PredictedClass) :-
    predict_value(X_value, root, PredictedClass).

% Predicción de valor basada en una característica del árbol
predict_value(_, Path, PredictedClass) :-
    feature(Path, feature(null, _, _,_, _, [Y_true, Y_false], _)),
    number(Y_true),
    number(Y_false),
    Y_true > Y_false,
    ground_truth(_, [PredictedClass, _]),
    !.

predict_value(_, Path, PredictedClass) :-
    feature(Path, feature(null, _, _,_, _, [Y_true, Y_false], _)),
    number(Y_true),
    number(Y_false),
    Y_true < Y_false,
    ground_truth(_, [_, PredictedClass]),
    !.

predict_value(X_value, Path, PredictedClass) :-
    feature(Path, feature(FeatureIndex, _, _, _, _, _, _)),
    key_in_dict(X_value, FeatureIndex, 0), % Asumiendo que 1 representa el valor True para la característica
    atom_concat(Path, '_left', LeftPath),
    predict_value(X_value, LeftPath, PredictedClass).

predict_value(X_value, Path, PredictedClass) :-
    feature(Path, feature(FeatureIndex, _, _, _, _, _, _)),
    key_in_dict(X_value, FeatureIndex, 1), % Asumiendo que 1 representa el valor True para la característica
    atom_concat(Path, '_right', RightPath),
    predict_value(X_value, RightPath, PredictedClass).

predict_value(X_value, Path, PredictedClass) :-
    feature(Path, feature(FeatureIndex, _, _, _, _, _, _)),
    key_in_dict(X_value, FeatureIndex, Num), % Asumiendo que 1 representa el valor True para la característica
    Num > 50,
    atom_concat(Path, '_right', RightPath),
    predict_value(X_value, RightPath, PredictedClass).

predict_value(X_value, Path, PredictedClass) :-
    feature(Path, feature(FeatureIndex, _, _, _, _, _, _)),
    key_in_dict(X_value, FeatureIndex, Num), % Asumiendo que 1 representa el valor True para la característica
    Num =< 50,
    atom_concat(Path, '_left', LeftPath),
    predict_value(X_value, LeftPath, PredictedClass).

% Obtener el valor asociado a una clave en un diccionario
key_in_dict(Dict, KeyToFind, Value) :-
    atom_string(KeyAtom, KeyToFind), % Convierte la cadena en átomo
    get_dict(KeyAtom, Dict, Value).
