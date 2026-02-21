from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 보험계</p><footer id='17' style='font-size:14px'>98</footer><p id='18' "
 "data-category='paragraph' style='font-size:16px'>약이 연장된 경우 연장된 날 기준으로 매년 현재의 "
 '예정기<br>초율(적용이율, 적용위험률, 부가보험요율) 적용 및 반려동<br>물의 연령 증가 등의 사유로 보험요율이 변동될 수 '
 "있으며<br>이 때의 보험료는「보험료 및 해약환급금 산출방법서」에<br>따라 산출합니다.</p><br><p id='19'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000369',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
