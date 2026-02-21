from langchain_core.documents import Document

chunk = Document(
    page_content=('. 관공서에서 수해, 화재나 그 밖의 재난을 조사하고 사망한 것으로 통보하는<br>경우: 가족관계등록부에 기재된 사망연월일을 기준으로 '
 "합니다.</p><br><p id='6' data-category='paragraph' style='font-size:14px'>부 가 설 "
 "명 실종선고</p><br><p id='7' data-category='paragraph' style='font-size:14px'>어떤 "
 "사람의 생사불명의 상태가 일정기간 이상 계속 될 때 이해관계가 있는 사</p><br><p id='8'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000354',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
