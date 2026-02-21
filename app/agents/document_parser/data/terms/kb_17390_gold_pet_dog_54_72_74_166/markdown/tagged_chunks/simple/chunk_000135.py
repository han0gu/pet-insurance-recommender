from langchain_core.documents import Document

chunk = Document(
    page_content=('- 니다.\n'
 '예 시| ∙ 보험나이 | 계산 |\n'
 '| --- | --- |\n'
 '| 생년월일 : : ⇒ 2022년 4월 1992년 10월 6월 = ∙ 계약해당일 최초계약일과 동일한 월, 일을 해당연도의 계약해당일이 '
 '없는 경우 에는 해당 월의 마지막 날을 합니다. 계약일: 2022년 10월 1일 => 10월 1일 계약일: 2024년 2월 29일 => '
 '2월 말일 | 1992년 10월 2일, 현재(계약일) 2022년 4월 13일 13일 - 2일 = 29년 11일 30세 계산 말합니다. '
 '계약해당일로 계약해당일: 계약해당일: |'),
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
 'indexing': {'chunk_id': 'chunk_000135',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
