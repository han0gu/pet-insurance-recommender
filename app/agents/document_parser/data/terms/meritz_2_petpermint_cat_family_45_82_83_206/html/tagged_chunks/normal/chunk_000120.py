from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 30일 이내에 승낙 또는 거절의 통지가 없으면<br>승낙된 것으로 봅니다.<br>\uf000 회사가 제1회 보험료를 받고 '
 '승낙을 거절한 경우에는 거<br>절통지와 함께 받은 금액을 계약자에게 돌려 드리며, 보험<br>료를 받은 기간에 대하여 평균공시이율 + '
 '1%를 연단위 복리<br>로 계산한 금액을 더하여 지급합니다'),
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
 'indexing': {'chunk_id': 'chunk_000120',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
