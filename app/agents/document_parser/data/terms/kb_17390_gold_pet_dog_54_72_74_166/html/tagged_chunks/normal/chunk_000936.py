from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 보험기간 중에 발생한 사고로 회사가 지급하는 연간 의료비보험금의 총<br>합계는 보험증권에 기재된 연간 총 보상한도액을 한도로 '
 '합니다.<br>\uf000 제1항에서 정한 의료비보험금은 보험증권에 기재된 자기부담금을 차감한 후 보험<br>증권에 기재된 보상비율을 '
 "곱한 금액이며 보험증권에 기재된 1일당 보상한도액을</p><br><p id='115' "
 "data-category='list'></p><br><h1 id='116' style='font-size:16px'>한도로 합니다"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000936',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
