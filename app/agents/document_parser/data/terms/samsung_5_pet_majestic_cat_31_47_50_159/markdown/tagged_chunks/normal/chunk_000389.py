from langchain_core.documents import Document

chunk = Document(
    page_content=('- 급사유 판정에 드는 의료비용은 회사가 전액 부담합니다.\n'
 '# <관련법규>[의료법 제3조(의료기관)에 규정한 종합병원]\n'
 '100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 전속하\n'
 '는 전문의를 둔 병원을 말합니다.# 제 4조 (보험금을 지급하지 않는 사유)회사는 아래의 사유를 원인으로 하여 생긴 손해는 보상하지 '
 '않습니다.- 1. 특별약관 일반사항 제7조(보험금을 지급하지 않는 사유) 제1항 제1호 및 제2호\n'
 '- 2. 피보험자의 배우자 및 직계존비속에 의한 손해'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000389',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
