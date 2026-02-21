from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험증권 등에 기재된 피보험자의 운전 목적이 변경된 경우<br>예) 자가용에서 영업용으로 변경, 영업용에서 자가용으로 변경 '
 '등<br>법<br>3. 보험증권 등에 기재된 피보험자의 운전여부가 변경된 경우<br>ㆍ<br>예) 비운전자에서 운전자로 변경, 운전자에서 '
 '비운전자로 변경 등<br>4'),
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
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 163,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
