from langchain_core.documents import Document

chunk = Document(
    page_content=('납입하지 않아 보험료 납입이 연<br>체중인 경우에 회사는 14일(보험기간이 1년 미만인 경우에는 7일) 이상의 '
 '기간을<br>납입최고(독촉)기간으로 정하여 계약자에게 다음 각 호의 내용을 서면(등기우편<br>등), 전화(음성녹음) 또는 전자문서 '
 '등으로 알려 드립니다'),
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
 'indexing': {'chunk_id': 'chunk_000880',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
