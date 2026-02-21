from langchain_core.documents import Document

chunk = Document(
    page_content=('도\n'
 '치과병원・한방병원・요양병원・정신병원・의원・치과의원・한의원 및 조산 성\n'
 '원으로 나누어집니다. 특\n'
 '약특수 다수인을 위하여 의제KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 87- 87 -제5조(특별약관의 소멸)\n'
 '피보험자가 사망하였을 경우에는 이 특별약관의 계약도 소멸되며 회사는 "보험료 및해약환급금 산출방법서"에서 정하는 바에 따라 피보험자의 '
 '사망 당시 이 특별약관의및 미경과보험료를 계약자에게 지급합니다.계약자적립액제6조(준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을 따릅니다. 다만,'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000378',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
