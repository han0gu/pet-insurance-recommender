from langchain_core.documents import Document

chunk = Document(
    page_content=('반려동물(강아지) 일반조항에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따\n'
 '릅니다. 다만, 보통약관 제1절 일반조항 제9조(만기환급금의 지급) 및 제36조(중도인출)은 제외합니다.# '
 '1.반려동물의료비Ⅱ(강아지)특\n'
 '제1조(보험금의 지급사유) 별\n'
 '\uf000 회사는 보험증권에 기재된 반려동물에게 이 특별약관의 보험기간 중 반려동물의료 약\n'
 '비의 보장개시일(이하 반려동물의료비보장개시일이라 합니다) 이후에 상해 또는 관\n'
 '질병(이하 사고라 합니다)이 발생하여 그 치료를 직접적인 목적으로 국내에서 수'),
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
 'indexing': {'chunk_id': 'chunk_000542',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
