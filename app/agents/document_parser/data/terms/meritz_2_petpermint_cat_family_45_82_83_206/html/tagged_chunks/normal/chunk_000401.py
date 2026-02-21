from langchain_core.documents import Document

chunk = Document(
    page_content=("추<br>심명령 또는 전부명령에 따라 회사는 채권자에게 해약환<br>급금을 지급하게 됩니다.</p><footer id='54' "
 "style='font-size:14px'>102</footer><p id='55' data-category='paragraph' "
 "style='font-size:20px'>또한, 국세 및 지방세 체납시 국세청 및 지방자치단체에<br>의해 채무자의 해약환급금이 압류될 "
 "수 있으며, 체납처<br>분 절차에 따라 회사는 채권자에게 해약환급금을 지급하<br>게 됩니다.</p><h1 id='56'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000401',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
