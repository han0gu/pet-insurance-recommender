from langchain_core.documents import Document

chunk = Document(
    page_content=('지, 무효, 취소 또는 철회에 따라 효력이 없어진 경우에는 이 특별약관은 그 때부터 효력\n'
 '이 없으며, 기본계약을 따릅니다.# 제2조 (용어의 정의)이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의 다른 조항에서 달리 '
 '정의되지\n'
 '않는 한 다음과 같습니다.# ① 계약관계 관련 용어- 1. 계약자: 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 '
 '말합니다.\n'
 '- 2. 보험수익자: 보험금 지급사유가 발생하는 때에 회사에 보험금을 청구하여 받을 수\n'
 '- 있는 사람을 말합니다.'),
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
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000127',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
