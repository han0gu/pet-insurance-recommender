from langchain_core.documents import Document

chunk = Document(
    page_content=('서 규정한 장애인인 보험# 【용어해설】く 「소득세법 시행령 제107조(장애인의 범위)」 에서 규정한 장애인>- 43 -당신에게 좋은보험 '
 '삼성화재- 1. 「장애인복지법」 에 따른 장애인 및 「장애아동 복지지원법」 에 따른 장애아동 중 기획재정부령으\n'
 '- 로 정하는 사람\n'
 '- 2. 「국가유공자 등 예우 및 지원에 관한 법률」 에 의한 상이자 및 이와 유사한 사람으로서 근로능력\n'
 '- 이 없는 사람\n'
 '- 3. 「국민건강보험법 시행령」 별표2 제3호 라목1)부터10)까지 외의 부분 전단에 따른 희귀성난치질환'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
