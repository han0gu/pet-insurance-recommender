from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 계약자<br>가 제1회 보험료를 신용카드로 납입한 계약의 승낙을 거절<br>하는 경우에는 신용카드의 매출을 취소하며 '
 "이자를 더하여<br>지급하지 않습니다.</p><p id='95' data-category='paragraph' "
 "style='font-size:20px'>제12조(특별약관의 무효)</p><br><p id='96' "
 "data-category='paragraph' style='font-size:20px'>\uf000 반려동물 비용손해 관련 특별약관을 "
 '체결할 때 이 특별<br>약관에서 정한 피보험자 및 반려동물의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000347',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
