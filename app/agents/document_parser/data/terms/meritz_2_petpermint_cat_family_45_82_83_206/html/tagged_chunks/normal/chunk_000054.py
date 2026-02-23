from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 같<br>습니다)에 대한 적립이율은 [보장]공시이율로 합니다.<br>\uf000 [보장]공시이율은 매월 마지막 날(다만, 마지막 '
 '날이 영<br>업일이 아닌 때에는 직전의 영업일로 함) 이전에 산출하며,<br>그 다음달에 한하여 적용합니다.<br>\uf000 회사는 '
 '이 계약의 사업방법서에서 정하는 바에 따라 운<br>용자산이익률과 외부지표금리수익률을 고려하여 산출된 공<br>시기준이율에 조정률을 '
 '반영하여 [보장]공시이율을 결정합<br>니다.<br>\uf000 [보장]공시이율의 최저보증이율은 연복리 0.3%로 '
 '합니<br>다.<br>\uf000 회사는 제1항부터'),
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
 'indexing': {'chunk_id': 'chunk_000054',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
