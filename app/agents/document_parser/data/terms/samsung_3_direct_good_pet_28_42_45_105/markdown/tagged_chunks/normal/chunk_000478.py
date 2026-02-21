from langchain_core.documents import Document

chunk = Document(
    page_content=('사이에 피보험자가 법률상의 배상책임을 부담함으로써 입은 손해에 대한 위험을 보장하\n'
 '기 위하여 체결됩니다.# 제 2조 (용어의 정의)이 특별약관에서 사용되는 용어의 정의는 이 특별약관의 다른 조항에서 달리 정의되지\n'
 '않는 한 다음과 같습니다.# ① 계약관계 관련 용어1. 피보험자 : 보험사고로 인하여 타인에 대한 법률상 손해배상책임을 부담하는 손해\n'
 '를 입은 사람(법인인 경우에는 그 이사 또는 법인의 업무를 집행하는 그 밖의 기관'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000478',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
