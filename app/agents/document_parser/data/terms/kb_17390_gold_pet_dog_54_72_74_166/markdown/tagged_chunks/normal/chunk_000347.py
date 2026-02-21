from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써【별표4】(골\n'
 '절분류표Ⅱ(치아파절제외))에서 정한 골절(치아의 파절(깨짐, 부러짐) 제외)로 진\n'
 '단확정 되고 그 치료를 직접적인 목적으로 "부목(Splint Cast)치료"(이하 부목치료\n'
 '라 합니다)를 받은 경우 이 특별약관의 보험가입금액을 골절부목치료비로 보험수익자에게 매 사고시마다 지급합니다.# 제2조(보험금 지급에 '
 '관한# 세부규정)- \uf000 제1조(보험금의 지급사유)의 골절부목치료비는 같은 상해를 직접적인 원인으로'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000347',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
