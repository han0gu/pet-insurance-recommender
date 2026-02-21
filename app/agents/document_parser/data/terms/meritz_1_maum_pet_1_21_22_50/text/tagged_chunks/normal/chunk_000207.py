from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 장애인으로서 그 장애기간이 기재된 장애인증명서를 제1항 따라 회사에 제출한 때에\n'
 '는 그 장애기간 동안은 이를 다시 제출하지 않을 수 있습니다.\n'
 '④ 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 경우 계약자는 이를 회사에\n'
 '알리고 변경된 장애기간이 기재된 장애인증명서를 제출하여야 합니다.제3조(장애인전용보험으로의 전환)① 회사는 이 특별약관이 부가된 '
 '전환계약을「소득세법 제59조의4(특별세액공제) 제1항\n'
 '제1호」에 해당하는 장애인전용보험으로 전환하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
