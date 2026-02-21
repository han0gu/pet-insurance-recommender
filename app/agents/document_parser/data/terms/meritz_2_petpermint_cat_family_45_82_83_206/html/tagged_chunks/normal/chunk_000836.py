from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 각각의 질병의 상태<br>등에 대한 수의사의 소견에 따라 다르게 적용할 수 있습니</p><footer id='88' "
 "style='font-size:14px'>166</footer><p id='0' data-category='paragraph' "
 "style='font-size:16px'>다.<br>\uf000 제2항에도 불구하고 보험업법 제97조 제1항 제5호 및 동<br>법 "
 '시행령 제43조의2 제1항에 따른 보장내용 등이 비슷한<br>보험계약(이하 「유사계약」이라 합니다)이 계약 청약일 현<br>재 '
 '유지중이거나, 계약 청약일 전'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000836',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
