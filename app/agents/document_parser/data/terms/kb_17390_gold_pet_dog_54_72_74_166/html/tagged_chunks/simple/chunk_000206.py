from langchain_core.documents import Document

chunk = Document(
    page_content=("id='18' style='font-size:16px'><thead><tr><td>∙ "
 '보험나이</td><td>계산</td></tr></thead><tbody><tr><td>생년월일 : : ⇒ 2022년 4월 1992년 '
 '10월 6월 = ∙ 계약해당일 최초계약일과 동일한 월, 일을 해당연도의 계약해당일이 없는 경우 에는 해당 월의 마지막 날을 합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000206',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
