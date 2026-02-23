from langchain_core.documents import Document

chunk = Document(
    page_content=('험계약일」이라 합니다)부터 그 날을 포함하여 30일이 지난 날의 다음날로 합니다. 이\n'
 '경우 보험계약일은 이 특별약관의 제1회 보험료를 받은 날로 합니다.- \n'
 '# <예시안내># [「반려견 사망위로금」에 대한 보장개시일(책임개시일) 계산]![image](/image/placeholder)\n'
 '보험계약일 보장개시일(책임개시일)\n'
 '◄───── 30일 ─────►'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000465',
              'chunk_char_len': 196,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
