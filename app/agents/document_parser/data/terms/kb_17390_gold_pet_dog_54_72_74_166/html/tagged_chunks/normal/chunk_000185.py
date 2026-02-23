from langchain_core.documents import Document

chunk = Document(
    page_content=(". 가. 서명자의 신원</td></tr></tbody></table><br><h1 id='1' "
 "style='font-size:14px'>나. 서명자가 해당 전자문서에 서명하였다는 사실</h1><p id='2' "
 "data-category='paragraph' style='font-size:14px'>제21조(계약의 무효)<br>다음 중 한 가지에 "
 '해당되는 경우에는 계약을 무효로 하며 이미 납입한 보험료를 돌<br>려드립니다'),
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
 'indexing': {'chunk_id': 'chunk_000185',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
