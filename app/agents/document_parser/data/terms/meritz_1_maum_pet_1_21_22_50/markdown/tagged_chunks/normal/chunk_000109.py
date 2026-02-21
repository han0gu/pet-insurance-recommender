from langchain_core.documents import Document

chunk = Document(
    page_content=('- 금액을 더하여 지급합니다.\n'
 '【설명】보험사가 해지권을 행사하는 경우 위의 ‘청구일’은 보험사의 해지 의사표시\n'
 '(서면, 전자우편, 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시 포함)가 보험\n'
 '계약자 또는 그의 대리인에게 도달한 날로 봅니다.제7관 분쟁의 조정 등# 제34조(분쟁의 조정)- ① 계약에 관하여 분쟁이 있는 경우 '
 '분쟁 당사자 또는 기타 이해관계인과 회사는 금융감독\n'
 '- 원장에게 조정을 신청할 수 있으며, 분쟁조정 과정에서 계약자는 관계 법령이 정하는'),
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
 'indexing': {'chunk_id': 'chunk_000109',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
