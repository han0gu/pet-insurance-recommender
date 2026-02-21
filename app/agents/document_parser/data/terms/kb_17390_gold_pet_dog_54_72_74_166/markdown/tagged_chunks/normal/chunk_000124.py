from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 보험종목\n'
 '- 2. 보험기간\n'
 '- 3. 보험료 납입방법 및 납입기간\n'
 '- 4. 계약자, 피보험자\n'
 '- 5. 보험가입금액, 보험료(적립보험료를 포함합니다), 배상책임의 경우 보상한도액\n'
 '| 수익자가 변경되었음을 | 회사에 통지하여야 합니다. |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000124',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
