from langchain_core.documents import Document

chunk = Document(
    page_content=('- 신장질환, 방광질환 및 각종결석의 대기기간은 90일 적용)으로 인한 손해. 단, 이 계약이 갱신\n'
 '- 계약인 경우에는 적용하지 않습니다.\n'
 '8. 수렵, 투견, 경주 등과 수색, 마약탐지, 경계등의 특수목적으로 업무수행 및 훈련 중의 손해\n'
 '9. 애완동물의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할 수 있는 증\n'
 '상을 포함합니다. 다만 보험기간 중 최초로 발견된 경우에는 당해 보험기간에 한하여 보상합니\n'
 '다.)- 10. 대한민국 이외 지역에서 발생한 사고 및 손해'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000013',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
