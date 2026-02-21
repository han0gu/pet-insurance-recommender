from langchain_core.documents import Document

chunk = Document(
    page_content=('. 관공서에서 수해, 화재나 그 밖의 재난을 조사하고 사망한 것으로 통보하는 '
 "경</td></tr></tbody></table><br><h1 id='21' style='font-size:16px'>우 : "
 "가족관계등록부에 기재된 사망연월일을 기준으로 합니다.</h1><br><table id='22' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>부 가 설 "
 '명</td><td>실종선고</td><td>특별</td></tr><tr><td colspan="3">어떤 사람의 생사불명의 상태가'),
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
 'indexing': {'chunk_id': 'chunk_000212',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
