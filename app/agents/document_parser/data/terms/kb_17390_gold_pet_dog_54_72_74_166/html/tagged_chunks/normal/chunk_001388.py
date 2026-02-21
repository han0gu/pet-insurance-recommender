from langchain_core.documents import Document

chunk = Document(
    page_content=('보장특약이 자동 갱신 물<br>된 경우에는 갱신보장특약의 제1회 보험료를 갱신 일까지 납입하여야 합니다.<br>\uf000 제1항에도 '
 '불구하고 계약자가 갱신 일까지 갱신보장특약의 제1회 보험료를 납입하<br>지 않는 때에는 보통약관 제1절 일반조항 제28조(보험료의 '
 '납입이 연체되는 경우<br>제<br>납입최고(독촉)와 계약의 해지)에 따라 납입최고(독촉)하며, 이 납입최고(독촉)<br>도<br>기간 '
 '안에 보험료를 납입하지 않는 경우 납입최고(독촉)기간이 끝나는 날의 다음<br>날 해당 보장특약은 해제된 것으로 봅니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001388',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
