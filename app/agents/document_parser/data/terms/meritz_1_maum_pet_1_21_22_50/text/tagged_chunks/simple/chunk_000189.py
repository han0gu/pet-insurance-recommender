from langchain_core.documents import Document

chunk = Document(
    page_content=('및 보험수익(이하「수익자」라 합니다)자가 모두 동일한 보통약관 및 특별약관에 적용됩니\n'
 '다.제2조(특별약관의 체결 및 소멸)① 이 특약은 계약자의 청약과 보험회사(이하 보험회사는 「회사」라 합니다)의 승낙으로\n'
 '부가되어집니다.\n'
 '② 제1조(적용대상)의 보험계약이 해지 또는 기타 사유에 의하여 효력을 가지지 않게 되는\n'
 '경우에는 이 특약은 더 이상 효력을 가지지 않습니다.제3조(지정대리청구인의 지정)① 보험계약자는 보통약관 또는 특별약관에서 정한 보험금을 '
 '직접 청구할 수 없는 특별한'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000189',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
