from langchain_core.documents import Document

chunk = Document(
    page_content=('② 보험수익자가 고의로 피보험자를 해친 경우. 다만, 그 보험수익자가 보험금의 일부 보험수익자인 경우에는 다른 보험수익자에 대한 보험금은 '
 '지급합니다. ③ 계약자가 고의로 피보험자를 해친 경우 ④ 피보험자의 임신, 출산(제왕절개를 포함합니다), 산후 기. 그러나 회사가 '
 '보장하는 보험금 지급사유와 보장 개시일부터 2년이 지난 후에 발생한 습관성 유산, 불 임 및 인공수정 관련 합병증으로 인한 경우에는 보험 '
 '금을 지급합니다.\n'
 '【습관성 유산, 불임 및 인공수정】\n'
 '한국표준질병·사인분류상의 N96~N98에 해당하는 질병을 말합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 51},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000020',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
