from langchain_core.documents import Document

chunk = Document(
    page_content=('지정대리청구서비스 특별약관\n'
 '제1조(적용대상)\n'
 '이 특별약관(이하「특약」이라 합니다)은 보험계약자(이하「계약자」라 합니다), 피보험자 및 보험수익(이하「수익자」라 합니다)자가 모두 '
 '동일한 보통약관 및 특별약관에 적용됩니 다.\n'
 '제2조(특별약관의 체결 및 소멸)\n'
 '① 이 특약은 계약자의 청약과 보험회사(이하 보험회사는 「회사」라 합니다)의 승낙으로 부가되어집니다. ② 제1조(적용대상)의 보험계약이 '
 '해지 또는 기타 사유에 의하여 효력을 가지지 않게 되는 경우에는 이 특약은 더 이상 효력을 가지지 않습니다.\n'
 '제3조(지정대리청구인의 지정)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 42},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000227',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
