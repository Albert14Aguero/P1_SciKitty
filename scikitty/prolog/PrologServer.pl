:- use_module(library(http/thread_httpd)).
:- use_module(library(http/http_dispatch)).
:- use_module(library(http/http_json)).
:- use_module(library(http/http_log)).
:- use_module(library(http/http_cors)).
:- use_module(library(http/html_write)).
:- [rulesgenerator].

% URL handlers.
:- http_handler('/compile', handle_request_compile, [method(post), time_limit(infinite), chunked]).  % Añadir time_limit y chunked para grandes cuerpos
:- http_handler('/predict', handle_request_predict, [method(post), time_limit(infinite), chunked]).  % Añadir time_limit y chunked para grandes cuerpos

handle_request_compile(Request) :-
    catch(
        ( http_read_json_dict(Request, Query),
          json_to_prolog_rules(Query),
          reply_json_dict(_{status: 'success'})
        ),
        Error,
        ( format('Content-type: text/plain~n~n'),
          format('Error: ~w~n', [Error]),
          format('Error processing JSON request.')
        )
    ).

handle_request_predict(Request) :-
    catch(
        ( http_read_json_dict(Request, Query),
          predict(Query, Class),
          reply_json_dict(_{class: Class})
        ),
        Error,
        ( format('Content-type: text/plain~n~n'),
          format('Error: ~w~n', [Error]),
          format('Error processing JSON request.')
        )
    ).

server(Port) :-
    http_server(http_dispatch, [port(Port)]).

set_setting(http:logfile, 'service_log_file.log').

:- initialization
    format('*** Starting Server ***~n', []),
    (current_prolog_flag(argv, [SPort | _]) -> true ; SPort='8000'),
    atom_number(SPort, Port),
    format('*** Serving on port ~d *** ~n', [Port]),
    set_setting_default(http:cors, [*]), % Allows cors for every
    server(Port).
