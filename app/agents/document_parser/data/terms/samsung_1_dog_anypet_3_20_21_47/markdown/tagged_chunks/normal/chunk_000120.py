from langchain_core.documents import Document

chunk = Document(
    page_content=('- 수 있습니다. 그러나 회사는 피보험자가 그 사고에 관하여 가지는 항변으로써 피해자에게 대항할\n'
 '- 수 있습니다.\n'
 '- ② 회사가 제1항의 청구를 받았을 때에는 지체없이 피보험자에게 통지하여야 하며, 회사의 요구가 있\n'
 '- 으면 계약자 및 피보험자는 필요한 서류증거의 제출, 증언 또는 증인출석에 협조하여야 합니다.\n'
 '- 27 -당신에게 좋은보험 삼성화재- ③ 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 필요하다고 인정할 때에는 피'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000120',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
