from langchain_core.documents import Document

chunk = Document(
    page_content=('- 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.# 제 4조 (특별약관의 무효)다음 중 한 가지에 해당되는 '
 '경우에는 이 특별약관은 무효로 하며 이미 납입한 이 특별약\n'
 '관의 보험료를 돌려 드립니다. 다만, 회사의 고의 또는 과실로 특별약관이 무효로 된 경\n'
 '우와 회사가 승낙 전에 무효임을 알았거나 알 수 있었음에도 불구하고 보험료를 반환하\n'
 '지 않은 경우에는 보험료를 납입한 날의 다음날부터 반환일까지의 기간에 대하여 회사는'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000471',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
