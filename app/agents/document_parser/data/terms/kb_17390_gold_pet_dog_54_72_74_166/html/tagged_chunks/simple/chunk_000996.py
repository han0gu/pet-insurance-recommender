from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반려동물의료비Ⅱ(강아지) 특별</p><br><p id='199' data-category='list' "
 "style='font-size:14px'>약관」에서 보상하는 의료비보험금 합계를 말합니다.<br>\uf000 제1항에서 정한 "
 '주요치료보험금은 제1항의 의료비에서 제2항의 반려동물의료비보<br>험금 및 보험증권에 기재된 자기부담금을 차감한 금액에 보험증권에 기재된 '
 '보상<br>비율을 곱한 금액이며, 보험증권에 기재된 치료구분별 각각의 지급한도 및 보상<br>한도액에 따라 보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000996',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
