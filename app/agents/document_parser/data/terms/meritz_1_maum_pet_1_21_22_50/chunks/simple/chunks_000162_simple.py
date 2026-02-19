from langchain_core.documents import Document

chunk = Document(
    page_content=('① 피보험자가 피해자에게 손해배상책임을 지는 사고가 생긴 때에는 피해자는 이 특별약관 에 따라 회사가 피보험자에게 지급책임을 지는 금액 '
 '한도내에서 회사에 대하여 보험금 의 지급을 직접 청구할 수 있습니다. 그러나 회사는 피보험자가 그 사고에 관하여 가지 는 항변으로써 '
 '피해자에게 대항할 수 있습니다. ② 회사는 제1항의 청구를 받았을 때에는 지체없이 피보험자에게 통지하여야 하며, 회사의 요구가 있으면 '
 '피보험자 및 계약자는 필요한 서류·증거의 제출, 증언 또는 증인 출석에 협조하여야 합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
