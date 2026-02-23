from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 피보험자가 피해자에게 손해배상책임을 지는 사고가 생긴 때에는 피해자는 이 특\n'
 '- 별약관에 따라 회사가 피보험자에게 지급책임을 지는 금액 한도내에서 회사에 대\n'
 '- 하여 보험금의 지급을 직접 청구할 수 있습니다. 그러나 회사는 피보험자가 그\n'
 '- 사고에 관하여 가지는 항변으로써 피해자에게 대항할 수 있습니다.\n'
 '- \uf000 회사는 제1항의 청구를 받았을 때에는 지체없이 피보험자에게 통지하여야 하며,\n'
 '- 회사의 요구가 있으면 피보험자 및 계약자는 필요한 서류・증거의 제출, 증언 또\n'
 '- 는 증인출석에 협조하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000686',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
