from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약일: 2022년 10월 1일 => 10월 1일 계약일: 2024년 2월 29일 => 2월 말일</td><td>1992년 10월 '
 '2일, 현재(계약일) 2022년 4월 13일 13일 - 2일 = 29년 11일 30세 계산 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 130,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
