from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여 90일 이내에 발생한 비뇨기계질환, 전염성복막염 또는 기타 이들과 '
 '유사한 질병 또는 상해에 대해서는 보험금을 지급하 지 않습니다. 단,「반려동물 비용손해 관련 특별약관 일반 조항」제15조(재가입) '
 '제6항에 따라 보험계약이 연장된 경 우에는 적용하지 않습니다. \uf000 제1항의「연간」이라 함은 계약일부터 매 1년 단위로 도 래하는 '
 '계약해당일 전일까지의 기간을 말합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 161},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000551',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
