from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서는 보통약관 제1절 일반조항 제9조(만기환급금의 지급), 제24조(계\n'
 '약의 소멸) 및 제36조(중도인출)는 제외합니다.상해흉터복원수술비Ⅱ(안면부)6.# 제1조(보험금의 지급사유)\uf000 회사는 피보험자가 '
 '이 특별약관의 보험기간 중에 급격하고도 우연한 외래의 사고\n'
 '로 병원 또는 의원(한방병원 또는 한의원을 포함합니다)등에서 치료를 받고 그\n'
 '직접적인 결과로 인하여 안면부에 외형상의 반흔(흉터)이나 추상장해, 신체의 기\n'
 '형이나 기능장해가 발생하여 그 원상회복을 목적으로 사고일로부터 2년 이내에'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000284',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
