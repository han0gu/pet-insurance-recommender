from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 않아 보험료 납입이중인 경우에 회사는 14일(보험기간이 1년 미만인 '
 '경우에는 7일) 이상의 기간을 납\n'
 '입최고(독촉)기간으로 정하여 계약자에게 다음 각 호의 내용을 서면(등기우편연체- 66 -- 등), 전화(음성녹음) 또는 전자문서 등으로 '
 '알려 드립니다. 다만, 해지 전에 발생\n'
 '- 한 보험금 지급사유에 대하여 회사는 보상하여 드립니다.\n'
 '- 1. 납입최고(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000160',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
