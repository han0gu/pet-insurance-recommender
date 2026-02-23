from langchain_core.documents import Document

chunk = Document(
    page_content=('- 109 -제KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 109도성특약|  | 예 시 의료비보험금의 계산 [의료비보험금 '
 '산출방식] {(피보험자가 부담한 1일당 의료비 – 1일당 자기부담금) X 보상비율}과 1일당 보상한도액 중 적은 금액 [의료비보험금 '
 '지급금액 예시] ·보상비율 70%, 자기부담금 3만원, 기본형Ⅱ 가입 기준 예시① 입/통원 중 수술을 하지 않은 날의 경우 ·피보험자가 '
 '부담한 당일 의료비 : 33만원 ·지급금액 = {(33만원 – 3만원) x 70%, 15만원} 중 적은 금액 = 15만원 예시② 입/통원'),
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
 'indexing': {'chunk_id': 'chunk_000548',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
