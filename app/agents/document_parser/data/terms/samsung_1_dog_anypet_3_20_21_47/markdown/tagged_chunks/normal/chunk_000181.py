from langchain_core.documents import Document

chunk = Document(
    page_content=('- 경된 장애기간이 기재된 장애인증명서를 제출하여야 합니다.\n'
 '- 44 -당신에게 좋은보험 삼성화재# 제3조(장애인전용보험으로의 전환)- ① 회사는 이 특약이 부가된 전환대상계약을 「소득세법 '
 '제59조의4(특별세액공제) 제1항 제1호」 에\n'
 '- 해당하는 장애인전용보험으로 전환하여 드립니다.\n'
 '- ② 제1항에 따라 전환대상계약이 장애인전용보험으로 전환된 후부터 납입된 전환대상계약 보험료는\n'
 '- 보험료 납입영수증에 장애인전용 보장성보험료로 표시됩니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
