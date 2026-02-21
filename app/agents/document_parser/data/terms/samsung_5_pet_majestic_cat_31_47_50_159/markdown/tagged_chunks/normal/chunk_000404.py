from langchain_core.documents import Document

chunk = Document(
    page_content=('100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 전속하\n'
 '는 전문의를 둔 병원을 말합니다.② 피보험자가 보험기간 중 사망하고, 그 후에「특정법정감염병」을 직접적인 원인으로\n'
 '사망한 사실이 확인된 경우에는 그 사망일을 진단 확정일로 보고 제1조(보험금의 지\n'
 '급사유)에 해당하는 경우에 한하여 해당 보험금을 지급합니다. 다만, 제5조(특별약관\n'
 '의 소멸)에 따라 이 특별약관의 계약자적립액 및 미경과보험료를 지급한 경우에는, 이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000404',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
