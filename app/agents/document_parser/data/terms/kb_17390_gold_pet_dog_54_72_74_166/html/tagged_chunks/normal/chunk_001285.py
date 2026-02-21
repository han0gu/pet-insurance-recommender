from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전<br>문의 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 "전액 부담</p><br><h1 id='121' style='font-size:14px'>합니다.</h1><p id='122' "
 "data-category='paragraph' style='font-size:14px'>제3조(보험금을 지급하지 않는 "
 '사유)<br>\uf000 회사는 보통약관 제1절 일반조항 제5조(보험금을 지급하지 않는 사유) 및 다음<br>중 어느 한 가지의 경우로 '
 '인하여 보험금 지급사유가 발생한'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001285',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
