from langchain_core.documents import Document

chunk = Document(
    page_content=('확인할 수 있습니다.특별약관 일반사항제1관 목적 및 용어의 정의# 제1조(목적)이 특별약관은 보험계약자(이하 "계약자"라 합니다)와 '
 '보험회사(이하 "회사"라 합니다) 사\n'
 '이에 피보험자의 질병이나 상해에 대한 위험을 보장하기 위하여 이 특별약관을 부가할\n'
 '수 있는 계약(이하 "기본계약"이라 합니다)에 부가하여 체결됩니다. 다만, 기본계약이 해\n'
 '지, 무효, 취소 또는 철회에 따라 효력이 없어진 경우에는 이 특별약관은 그 때부터 효력'),
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
 'indexing': {'chunk_id': 'chunk_000126',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
