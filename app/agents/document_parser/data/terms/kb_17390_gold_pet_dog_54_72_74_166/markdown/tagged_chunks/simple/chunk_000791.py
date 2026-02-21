from langchain_core.documents import Document

chunk = Document(
    page_content=('- 니다. 이하 "보통약관"이라 합니다)을 체결할 때 계약자의 청약과 회사의 승낙으\n'
 '- 로 보통약관에 부가하여 이루어집니다.\n'
 '- \uf000 이 특별약관을 통하여 전자서명법 제2조 제2호에 따른 전자서명으로 계약을 청약할\n'
 '- 수 있으며, 이 경우 전자서명은 자필서명과 동일한 효력을 갖는 것으로 합니다.\n'
 '- 제3조(약관교부의\n'
 '특례)\uf000 계약자가 원하는 방법에 따라 상품설명서, 보험약관 및 계약자 보관용 청약서 등- (보험증권은 제외하며, 이하 "보험계약 '
 '안내자료"라 합니다)을 전자우편 및 전자'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000791',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
