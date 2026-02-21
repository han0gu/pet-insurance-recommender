from langchain_core.documents import Document

chunk = Document(
    page_content=('상해흉터복원수술비 수술 1cm당 14만원\n'
 '(단, 3cm이상의 경우에 한함)\n'
 '주) 길이측정이 불가한 피부이식수술 등의 경우 수술cm는 최장직경으로 함 제\uf000 제1항에서 정한 안면부, 상지, 하지란 다음을 '
 '말합니다. 도\n'
 '성\n'
 '1. 안면부란 이마를 포함하여 목까지의 얼굴부분을 말합니다.\n'
 '특\n'
 '2. 상지란 견관절 이하의 팔부분을 말합니다.\n'
 '약\n'
 '3. 하지란 고관절 이하 대퇴부, 하퇴부, 족부를 의미하며, 둔부, 서혜부, 복부 등KB 금쪽같은 펫보험(강아지)(무배당)(26.01) '
 '77- 77 -관은 제외합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'joint', 'skin']},
 'indexing': {'chunk_id': 'chunk_000275',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
