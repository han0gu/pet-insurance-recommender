from langchain_core.documents import Document

chunk = Document(
    page_content=('# 【한국표준질병사인분류 부호 체계】질병의 원인과 증상 두 가지 모두에 관한 정보를 포함\n'
 '하는 진단을 위해 아래 두 가지 분류부호가 사용됩니\n'
 '다. 또한 원인과 질환에 따라 동시에 사용될 수 있습니\n'
 '다.- - 검표(+) : 원인이 되는 질환에 대한 질병분류코드\n'
 '- - 별표(*) : 원인(검표)으로 인한 발현증세에 대한 질\n'
 '- 병분류코드\n'
 '\uf000 지급금과 이자율 관련 용어| 용어 | 정의 |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000006',
              'chunk_char_len': 229,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
