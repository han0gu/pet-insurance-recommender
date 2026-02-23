from langchain_core.documents import Document

chunk = Document(
    page_content=(". 또한, 보장개시일을 계약일로 봅니다.</p><p id='29' data-category='paragraph' "
 "style='font-size:20px'>제17조(보험료의 납입이 연체되는 경우 납입최고(독촉)와<br>계약의 해지)</p><br><p "
 "id='30' data-category='paragraph' style='font-size:20px'>\uf000 계약자가 제2회 이후의 "
 '보험료를 납입기일까지 납입하지<br>않아 보험료 납입이 연체 중인 경우에 회사는 14일(보험기<br>간이 1년 미만인 경우에는 7일) '
 '이상의 기간을'),
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
 'indexing': {'chunk_id': 'chunk_000380',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
