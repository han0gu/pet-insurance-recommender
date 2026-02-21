from langchain_core.documents import Document

chunk = Document(
    page_content=('하여 갱신되는 것으로 하며, 세부사항은 회사의 사업방법서를 따릅니다.\n'
 '다. 동일한 사고에 대하여 갱신전 계약에서 이미 보험금 지급사유가 발생하여 해당\n'
 '보험금이 지급된 경우에는 갱신계약에서 보상하지 않습니다.- \n'
 '# 제4조 (갱신계약 제1회 보험료의 납입최고(독촉)와 갱신계약의 해제)① 계약자가 갱신계약의 제1회 보험료를 납입기일까지 납입하지 않은 '
 '때에는 보통약관\n'
 '제30조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따라 납입최\n'
 '고(독촉)하며, 이 납입최고(독촉)기간 안에 보험료가 납입되지 않은 경우 납입최고(독'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000696',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
