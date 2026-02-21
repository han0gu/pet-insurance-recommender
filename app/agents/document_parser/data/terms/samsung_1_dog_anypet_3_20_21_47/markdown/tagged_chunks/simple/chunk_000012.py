from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 이들의 특성에 의한 사고로 생긴 손해. 그리고, 위 이외의 방사선 조사(照射) 또는 방사능 오\n'
 '- 염으로 인한 손해.\n'
 '- 6. 보험개시일 이전에 이미 감염 또는 발병한 질병 및 상해. 갱신계약의 경우 최초 보험개시일은\n'
 '- 최초 보험가입시점 이후를 말합니다. 단, 보험종기와 갱신계약의 보험시기 사이에 일시적으로\n'
 '- 계약체결이 중단된 기간의 사고에 의한 경우에는 보험금을 지급하지 않습니다.\n'
 '- 7. 보험개시일로부터 30일 이내(이하"대기기간")에 발생한 질병(단, 암, 백내장, 녹내장, 심장질환,'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000012',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
