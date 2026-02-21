from langchain_core.documents import Document

chunk = Document(
    page_content=('중에 발생한 손해에 대해서는 보상하지 않습니다.# 제2조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관 및 해당 특별약관을 '
 '따릅니다.- 46 -당신에게 좋은보험 삼성화재# 펫샵 전용 전염병 보장 특별약관# 제1조(보상하는 손해)1 회사는 보통약관 '
 '제5조(보상하지 않는 손해) 제2항 제16호에도 불구하고, 펫샵에서 반려동물을 분\n'
 '양받은 후 가입하는 계약에 한하여 보험개시일로부터 30일 이내에 아래 각호의 질병이 발병함으로\n'
 '인하여 발생한 손해를 보상하여 드립니다. 단, 계약자 또는 피보험자의 명백하고 의도적인 태만으'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000189',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
